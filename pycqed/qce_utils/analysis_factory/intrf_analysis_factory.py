# -------------------------------------------
# Module containing interface for analysis factory components.
# -------------------------------------------
from abc import ABC, abstractmethod, ABCMeta
from dataclasses import dataclass, field
from typing import TypeVar, Dict, Type, List, Generic, Union
import logging
from enum import Enum, unique
import matplotlib.pyplot as plt
from pycqed.qce_utils.custom_exceptions import (
    InterfaceMethodException,
)


# Set up basic configuration for logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s:%(message)s')


T = TypeVar('T', bound=Type)


@dataclass(frozen=True)
class FigureDetails:
    figure_object: plt.Figure
    identifier: str


class IFactoryManager(ABC, Generic[T], metaclass=ABCMeta):
    """
    Interface class, describing methods for manager factories.
    """

    # region Class Methods
    @abstractmethod
    def analyse(self, response: T) -> List[FigureDetails]:
        """
        Constructs one or multiple (matplotlib) figures from characterization response.
        :param response: Characterization response used to construct analysis figures.
        :return: Array-like of analysis figures.
        """
        raise InterfaceMethodException
    # endregion
