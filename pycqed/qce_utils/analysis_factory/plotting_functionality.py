# -------------------------------------------
# General plotting functionality.
# -------------------------------------------
from abc import abstractmethod
from collections.abc import Iterable as ABCIterable
from typing import Callable, Tuple, Optional, Iterable, List, Union
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum
from pycqed.qce_utils.custom_exceptions import InterfaceMethodException

IFigureAxesPair = Tuple[plt.Figure, plt.Axes]
KEYWORD_LABEL_FORMAT = 'label_format'
KEYWORD_AXES_FORMAT = 'axes_format'
KEYWORD_HOST_AXES = 'host_axes'


class IAxesFormat:
    """
    Interface for applying formatting changes to axis.
    """
    # region Interface Methods
    @abstractmethod
    def apply_to_axes(self, axes: plt.Axes) -> plt.Axes:
        """
        Applies axes formatting settings to axis.
        :param axes: Axes to be formatted.
        :return: Updated Axes.
        """
        raise InterfaceMethodException
    # endregion

    # region Static Class Methods
    @staticmethod
    @abstractmethod
    def default() -> 'IAxesFormat':
        """:return: Default formatting instance."""
        raise InterfaceMethodException
    # endregion


class LabelFormat(IAxesFormat):
    """
    Specifies callable formatting functions for both vector components.
    """
    IFormatCall = Callable[[float], str]
    _default_format: IFormatCall = lambda x: f'{round(x)}'
    _default_label: str = 'Default Label [a.u.]'
    _default_symbol: str = 'X'

    # region Class Properties
    @property
    def x_label(self) -> str:
        """:return: Unit label for x-vector component."""
        return self._x_label

    @property
    def y_label(self) -> str:
        """:return: Unit label for y-vector component."""
        return self._y_label

    @property
    def z_label(self) -> str:
        """:return: Unit label for z-vector component."""
        return self._z_label

    @property
    def x_format(self) -> IFormatCall:
        """:return: Formatting function of x-vector component."""
        return self._x_format

    @property
    def y_format(self) -> IFormatCall:
        """:return: Formatting function of y-vector component."""
        return self._y_format

    @property
    def z_format(self) -> IFormatCall:
        """:return: Formatting function of z-vector component."""
        return self._z_format

    @property
    def x_symbol(self) -> str:
        """:return: Unit symbol for x-vector component."""
        return self._x_symbol

    @property
    def y_symbol(self) -> str:
        """:return: Unit symbol for y-vector component."""
        return self._y_symbol

    @property
    def z_symbol(self) -> str:
        """:return: Unit symbol for z-vector component."""
        return self._z_symbol
    # endregion

    # region Class Constructor
    def __init__(
            self,
            x_label: str = _default_label,
            y_label: str = _default_label,
            z_label: str = _default_label,
            x_format: IFormatCall = _default_format,
            y_format: IFormatCall = _default_format,
            z_format: IFormatCall = _default_format,
            x_symbol: str = _default_symbol,
            y_symbol: str = _default_symbol,
            z_symbol: str = _default_symbol
    ):
        self._x_label: str = x_label
        self._y_label: str = y_label
        self._z_label: str = z_label
        self._x_format: LabelFormat.IFormatCall = x_format
        self._y_format: LabelFormat.IFormatCall = y_format
        self._z_format: LabelFormat.IFormatCall = z_format
        self._x_symbol: str = x_symbol
        self._y_symbol: str = y_symbol
        self._z_symbol: str = z_symbol
    # endregion

    # region Interface Methods
    def apply_to_axes(self, axes: plt.Axes) -> plt.Axes:
        """
        Applies label formatting settings to axis.
        :param axes: Axes to be formatted.
        :return: Updated Axes.
        """
        axes.set_xlabel(self.x_label)
        axes.set_ylabel(self.y_label)
        if hasattr(axes, 'set_zlabel'):
            axes.set_zlabel(self.z_label)
        return axes
    # endregion

    # region Static Class Methods
    @staticmethod
    def default() -> 'LabelFormat':
        """:return: Default LabelFormat instance."""
        return LabelFormat()
    # endregion


class AxesFormat(IAxesFormat):
    """
    Specifies general axis formatting functions.
    """

    # region Interface Methods
    def apply_to_axes(self, axes: plt.Axes) -> plt.Axes:
        """
        Applies axes formatting settings to axis.
        :param axes: Axes to be formatted.
        :return: Updated Axes.
        """
        axes.grid(True, alpha=0.5, linestyle='dashed')  # Adds dashed gridlines
        axes.set_axisbelow(True)  # Puts grid on background
        return axes
    # endregion

    # region Static Class Methods
    @staticmethod
    def default() -> 'AxesFormat':
        """:return: Default AxesFormat instance."""
        return AxesFormat()
    # endregion


class EmptyAxesFormat(AxesFormat):
    """
    Overwrites AxesFormat with 'null' functionality.
    Basically leaving the axes unchanged.
    """

    # region Interface Methods
    def apply_to_axes(self, axes: plt.Axes) -> plt.Axes:
        """
        Applies axes formatting settings to axis.
        :param axes: Axes to be formatted.
        :return: Updated Axes.
        """
        return axes
    # endregion


class SubplotKeywordEnum(Enum):
    """
    Constructs specific enumerator for construct_subplot() method accepted keyword arguments.
    """
    LABEL_FORMAT = 'label_format'
    AXES_FORMAT = 'axes_format'
    HOST_AXES = 'host_axes'
    PROJECTION = 'projection'
    FIGURE_SIZE = 'figsize'


# TODO: Extend (or add) functionality to construct mosaic plots
def construct_subplot(*args, **kwargs) -> IFigureAxesPair:
    """
    Extends plt.subplots() by optionally working from host_axes
    and applying label- and axes formatting.
    :param args: Positional arguments that are passed to plt.subplots() method.
    :param kwargs: Key-word arguments that are passed to plt.subplots() method.
    :keyword label_format: (Optional) Formatting settings for figure labels.
    :keyword axes_format: (Optional) Formatting settings for figure axes.
    :keyword host_axes: (Optional) figure-axes pair to which to write the plot instead.
        If not supplied, create a new figure-axes pair.
    :return: Tuple of plotted figure and axis.
    """
    # Kwarg retrieval
    label_format: IAxesFormat = kwargs.pop(SubplotKeywordEnum.LABEL_FORMAT.value, LabelFormat.default())
    axes_format: IAxesFormat = kwargs.pop(SubplotKeywordEnum.AXES_FORMAT.value, AxesFormat.default())
    host_axes: Tuple[plt.Figure, plt.Axes] = kwargs.pop(SubplotKeywordEnum.HOST_AXES.value, None)
    projection: Optional[str] = kwargs.pop(SubplotKeywordEnum.PROJECTION.value, None)

    # Figure and axis
    if host_axes is not None:
        fig, ax0 = host_axes
    else:
        fig, ax0 = plt.subplots(*args, **kwargs)

    # region Dress Axes
    axes: Iterable[plt.Axes] = [ax0] if not isinstance(ax0, ABCIterable) else ax0
    for _ax in axes:
        _ax = label_format.apply_to_axes(axes=_ax)
        _ax = axes_format.apply_to_axes(axes=_ax)
    # endregion

    return fig, ax0


def draw_object_summary(host: IFigureAxesPair, params: object, apply_tight_layout: bool = True) -> IFigureAxesPair:
    """
    Adds text window with fit summary based on model parameter.
    :param host: Tuple of figure and axis.
    :param params: Any object (or model parameter container class) that implements .__str__() method.
    :param apply_tight_layout: (Optional) Boolean, whether a tight layout call should be applied to figure.
    :return: Tuple of plotted figure and axis.
    """

    def linebreaks_to_columns(source: List[str], column: int, column_spacing: int) -> str:
        """
        Attempts to insert tab spacing between source elements to create the visual illusion of columns.
        :param source: Array-like of string elements to be placed in column-like structure.
        :param column: Integer number of (maximum) columns.
        :param column_spacing: Integer column spacing in character units.
        :return: Single string with tabs to create column-like behaviour.
        """
        # Data allocation
        source_count: int = len(source)
        desired_count: int = -(source_count // -column) * column  # 'Upside down' floor division.
        pad_count: int = desired_count - source_count
        padded_source: List[str] = source + [''] * pad_count
        slice_idx: List[Tuple[int, int]] = [(i * column, (i + 1) * column) for i in range(desired_count // column)]
        result: str = ''
        for i, (lb, ub) in enumerate(slice_idx):
            row_elems = padded_source[lb: ub]
            linebreak: str = '' if i == len(slice_idx) - 1 else '\t\n'  # Only linebreak if there is another line coming
            result += ('\t'.join(row_elems) + linebreak).expandtabs(tabsize=column_spacing)
        return result

    fig, ax0 = host
    text_str: str = params.__str__()
    fontsize: int = 10
    ax0.text(
        x=1.05,
        y=0.99,
        s=text_str,
        fontdict=dict(horizontalalignment='left'),
        transform=ax0.transAxes,
        fontsize=fontsize,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor='#C5C5C5', alpha=0.5),
        linespacing=1.6,
    )
    if apply_tight_layout:
        fig.tight_layout()

    return fig, ax0
