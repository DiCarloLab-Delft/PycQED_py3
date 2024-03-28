# -------------------------------------------
# Factory module for constructing transmon-flux-arc identifier analysis.
# -------------------------------------------
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple
import warnings
import numpy as np
import matplotlib.transforms as transforms
from scipy.optimize import minimize
from pycqed.qce_utils.custom_exceptions import InterfaceMethodException
from pycqed.qce_utils.analysis_factory.intrf_analysis_factory import IFactoryManager, FigureDetails
from pycqed.qce_utils.analysis_factory.plotting_functionality import (
    construct_subplot,
    SubplotKeywordEnum,
    LabelFormat,
    AxesFormat,
    IFigureAxesPair,
)


@dataclass(frozen=True)
class Vec2D:
    """
    Data class, containing x- and y-coordinate vector.
    """
    x: float
    y: float

    # region Class Methods
    def to_vector(self) -> np.ndarray:
        return np.asarray([self.x, self.y])

    def to_tuple(self) -> Tuple[float, float]:
        return self.x, self.y

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> 'Vec2D':
        return Vec2D(
            x=vector[0],
            y=vector[1],
        )

    def __add__(self, other):
        if isinstance(other, Vec2D):
            return Vec2D(x=self.x + other.x, y=self.y + other.y)
        raise NotImplemented(f"Addition with anything other than {Vec2D} is not implemented.")
    # endregion


class IFluxArcIdentifier(ABC):
    """
    Interface class, describing properties and get-methods for flux-arc identifier.
    """

    @property
    @abstractmethod
    def polynomial(self) -> np.polyfit:
        """:return: Internally fitted polynomial."""
        raise InterfaceMethodException

    @property
    @abstractmethod
    def origin(self) -> Vec2D:
        """:return: (Flux) arc origin x-y 2D vector."""
        raise InterfaceMethodException

    @abstractmethod
    def get_amplitudes_at(self, detuning: float) -> np.ndarray:
        """
        Filters only real roots.
        :param detuning: detuning (y-value) at which to find the corresponding amplitude roots (x-values).
        :return: Amplitudes (x-values) corresponding to desired detuning (y-values).
        """
        roots: np.ndarray = (self.polynomial - detuning).roots
        return roots[np.isclose(roots.imag, 0)].real


@dataclass(frozen=True)
class FluxArcIdentifier(IFluxArcIdentifier):
    """
    Data class, containing (AC) flux pulse amplitude vs (Ramsey) frequency detuning.
    """
    _amplitude_array: np.ndarray = field(init=True)
    _detuning_array: np.ndarray = field(init=True)
    _polynomial: np.polyfit = field(init=False)

    # region Class Properties
    @property
    def amplitudes(self) -> np.ndarray:
        return self._amplitude_array

    @property
    def detunings(self) -> np.ndarray:
        return self._detuning_array

    @property
    def polynomial(self) -> np.polyfit:
        """:return: Internally fitted polynomial."""
        return self._polynomial

    @property
    def origin(self) -> Vec2D:
        """:return: (Flux) arc origin x-y 2D vector."""
        _polynomial = self.polynomial
        result = minimize(_polynomial, x0=0)
        return Vec2D(
            x=result.x[0],
            y=result.fun,
        )

    # endregion

    # region Class Methods
    def __post_init__(self):
        object.__setattr__(self, '_polynomial', self._construct_poly_fit(
            x=self.amplitudes,
            y=self.detunings,
        ))

    def get_amplitudes_at(self, detuning: float) -> np.ndarray:
        """
        Filters only real roots.
        :param detuning: detuning (y-value) at which to find the corresponding amplitude roots (x-values).
        :return: Amplitudes (x-values) corresponding to desired detuning (y-values).
        """
        roots: np.ndarray = (self.polynomial - detuning).roots
        real_roots: np.ndarray = roots[np.isclose(roots.imag, 0)].real
        if len(real_roots) == 0:
            warnings.warn(**PolynomialRootNotFoundWarning.warning_format(detuning))
        return real_roots
    # endregion

    # region Static Class Methods
    @staticmethod
    def _construct_poly_fit(x: np.ndarray, y: np.ndarray) -> np.poly1d:
        """:return: Custom polynomial a*x^4 + b*x^3 + c*x^2 + d*x + 0."""
        # Construct the design matrix including x^4, x^3, x^2, and x^1.
        x_stack = np.column_stack((x ** 4, x ** 3, x ** 2, x))
        # Perform the linear least squares fitting
        coefficients, residuals, rank, s = np.linalg.lstsq(x_stack, y, rcond=None)
        # coefficients are the coefficients for x^4, x^3, x^2, and x^1 term respectively
        a, b, c, d = coefficients
        return np.poly1d([a, b, c, d, 0])
    # endregion


class FluxArcIdentifierAnalysis(IFactoryManager[FluxArcIdentifier]):

    # region Class Methods
    def analyse(self, response: FluxArcIdentifier) -> List[FigureDetails]:
        """
        Constructs one or multiple (matplotlib) figures from characterization response.
        :param response: Characterization response used to construct analysis figures.
        :return: Array-like of analysis figures.
        """
        fig, ax = self.plot_flux_arc_identifier(
            identifier=response,
        )

        return [
            FigureDetails(figure_object=fig, identifier="voltage_to_detuning"),
        ]

    # endregion

    # region Static Class Methods
    @staticmethod
    def format_coefficient(coef):
        """Format coefficient into scientific notation with LaTeX exponent format."""
        return f"{coef:.2e}".replace('+0', '^{').replace('-0', '-') + '}'

    @staticmethod
    def plot_flux_arc_identifier(identifier: FluxArcIdentifier, **kwargs) -> IFigureAxesPair:
        """
        :param identifier:
        :param kwargs:
        :return:
        """
        # Data allocation
        nyquist_frequency: float = 1.3e9  # Based on AWG sampling rate of 2.4GHz
        roots: np.ndarray = identifier.get_amplitudes_at(detuning=nyquist_frequency)
        min_root: float = float(np.min(np.abs(roots)))
        high_resolution_amplitudes: np.ndarray = np.linspace(-min_root, min_root, 101)
        # Evaluate the fitted polynomial
        fitted_polynomial = identifier.polynomial
        y_fit = fitted_polynomial(high_resolution_amplitudes)
        origin: Vec2D = identifier.origin

        kwargs[SubplotKeywordEnum.LABEL_FORMAT.value] = kwargs.get(SubplotKeywordEnum.LABEL_FORMAT.value, LabelFormat(
            x_label='Output voltage [V]',
            y_label='Detuning [Hz]',
        ))
        fig, ax = construct_subplot(**kwargs)
        ax.plot(
            identifier.amplitudes,
            identifier.detunings,
            linestyle='none',
            marker='o',
        )
        ax.plot(
            high_resolution_amplitudes,
            y_fit,
            linestyle='--',
            marker='none',
            color='k',
        )
        ax.axhline(origin.y, linestyle='--', color='lightgrey', zorder=-1)
        ax.axvline(origin.x, linestyle='--', color='lightgrey', zorder=-1)

        # Display the polynomial equation in the plot
        a, b, c, d, _ = fitted_polynomial.coeffs
        formatter = FluxArcIdentifierAnalysis.format_coefficient
        equation_text = f"$y = {formatter(a)}x^4 + {formatter(b)}x^3 + {formatter(c)}x^2 + {formatter(d)}x$"
        ax.text(0.5, 0.95, equation_text, transform=ax.transAxes, ha='center', va='top')

        ylim = ax.get_ylim()
        # Draw horizontal line to indicate asymmetry
        desired_detuning: float = 500e6
        roots: np.ndarray = identifier.get_amplitudes_at(detuning=desired_detuning)
        if roots.size > 0:
            negative_root = float(roots[roots <= 0])
            negative_arc_x = negative_root
            negative_arc_y = fitted_polynomial(negative_arc_x)
            positive_arc_x = -negative_arc_x
            positive_arc_y = fitted_polynomial(positive_arc_x)
            # Draw comparison lines
            color: str = 'green'
            ax.hlines(y=negative_arc_y, xmin=min(high_resolution_amplitudes), xmax=origin.x, linestyle='--', color=color, zorder=-1)
            ax.hlines(y=positive_arc_y, xmin=origin.x, xmax=positive_arc_x, linestyle='--', color=color, zorder=-1)
            ax.vlines(x=negative_arc_x, ymin=ylim[0], ymax=negative_arc_y, linestyle='--', color=color, zorder=-1)
            # Draw annotations
            ax.annotate('', xy=(origin.x, positive_arc_y), xytext=(origin.x, negative_arc_y), arrowprops=dict(arrowstyle="<->", color=color))
            delta: float = abs(positive_arc_y - negative_arc_y)
            arrow_y_position: float = min(positive_arc_y, negative_arc_y) + delta / 2
            text_y_position: float = max(positive_arc_y * 1.05, arrow_y_position)
            ax.text(origin.x, text_y_position, f' $\Delta={delta * 1e-6:.2f}$ MHz', ha='left', va='center')
            ax.text(negative_arc_x, negative_arc_y, f' {desired_detuning * 1e-6:.0f} MHz at {negative_arc_x:.2f} V', ha='left', va='bottom')
        # Draw origin offset
        transform = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(0.98, origin.y, f'{origin.y * 1e-6:.3f} MHz', ha='right', va='bottom', transform=transform)

        ax.set_xlim(left=min(high_resolution_amplitudes), right=max(high_resolution_amplitudes))
        ax.set_ylim(ylim)
        return fig, ax
    # endregion
