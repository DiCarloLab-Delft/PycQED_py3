# -------------------------------------------
# Module containing interface for surface-code connectivity structure.
# -------------------------------------------
from abc import ABC, ABCMeta, abstractmethod
from typing import List, Union
from enum import Enum
from pycqed.qce_utils.custom_exceptions import InterfaceMethodException
from pycqed.qce_utils.control_interfaces.intrf_channel_identifier import (
    IQubitID,
    IEdgeID,
)
from pycqed.qce_utils.control_interfaces.intrf_connectivity import IDeviceLayer


class ParityType(Enum):
    STABILIZER_X = 0
    STABILIZER_Z = 1


class IParityGroup(ABC):
    """
    Interface class, describing qubit (nodes) and edges related to the parity group.
    """

    # region Interface Properties
    @property
    @abstractmethod
    def parity_type(self) -> ParityType:
        """:return: Parity type (X or Z type stabilizer)."""
        raise InterfaceMethodException

    @property
    @abstractmethod
    def ancilla_id(self) -> IQubitID:
        """:return: (Main) ancilla-qubit-ID from parity."""
        raise InterfaceMethodException

    @property
    @abstractmethod
    def data_ids(self) -> List[IQubitID]:
        """:return: (All) data-qubit-ID's from parity."""
        raise InterfaceMethodException

    @property
    @abstractmethod
    def edge_ids(self) -> List[IEdgeID]:
        """:return: (All) edge-ID's between ancilla and data qubit-ID's."""
        raise InterfaceMethodException
    # endregion

    # region Interface Methods
    @abstractmethod
    def contains(self, element: Union[IQubitID, IEdgeID]) -> bool:
        """:return: Boolean, whether element is part of parity group or not."""
        raise InterfaceMethodException
    # endregion


class ISurfaceCodeLayer(IDeviceLayer, metaclass=ABCMeta):
    """
    Interface class, describing surface-code relation based connectivity.
    """

    # region Interface Properties
    @property
    @abstractmethod
    def parity_group_x(self) -> List[IParityGroup]:
        """:return: (All) parity groups part of X-stabilizers."""
        raise InterfaceMethodException

    @property
    @abstractmethod
    def parity_group_z(self) -> List[IParityGroup]:
        """:return: (All) parity groups part of Z-stabilizers."""
        raise InterfaceMethodException
    # endregion

    # region Interface Methods
    @abstractmethod
    def get_parity_group(self, element: Union[IQubitID, IEdgeID]) -> IParityGroup:
        """:return: Parity group of which element (edge- or qubit-ID) is part of."""
        raise InterfaceMethodException
    # endregion
